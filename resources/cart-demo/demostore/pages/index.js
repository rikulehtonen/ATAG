import CategoryCard from '../components/CategoryCard';
import styles from '../styles/Home.module.css';

const HomePage = () => {
  return (
    <main className={styles.container}>
      <div className={styles.frontcat}>
        <CategoryCard image="/img/home.jpg" name="Home" />
        <CategoryCard image="/img/security.jpg" name="Security" />
        <CategoryCard image="/img/healtcare.jpg" name="Healthcare" />
        <CategoryCard image="/img/entertainment.jpg" name="Entertainment" />
      </div>
    </main>
  );
};

export default HomePage;

